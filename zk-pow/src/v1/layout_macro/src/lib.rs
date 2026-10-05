use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{braced, parse::Parse, parse_macro_input, punctuated::Punctuated, token::Mod, Ident, LitInt, Token};

#[proc_macro]
pub fn define_layout(input: TokenStream) -> TokenStream {
    let LayoutDef { module, items } = parse_macro_input!(input as LayoutDef);

    // Extract names & lengths
    let names: Vec<&Ident> = items.iter().map(|it| &it.name).collect();
    let lens: Vec<&LitInt> = items.iter().map(|it| &it.len).collect();

    // Build cumulative starts
    let mut starts = Vec::<usize>::new();
    let mut acc = 0usize;
    for lit in &lens {
        starts.push(acc);
        acc += lit.base10_parse::<usize>().unwrap();
    }
    let total = acc;

    // Generate constants per item
    let per_item_consts = names.iter().enumerate().map(|(idx, ident)| {
        let len = lens[idx];
        let start = starts[idx];
        let end = start + len.base10_parse::<usize>().unwrap();

        let len_ident = format_ident!("{}_LEN", ident);
        let start_ident = format_ident!("{}_START", ident);
        let end_ident = format_ident!("{}_END", ident);
        let range_ident = format_ident!("{}_RANGE", ident);

        quote! {
            pub const #ident       : usize = #start;  // Alias for _START
            pub const #len_ident   : usize = #len;
            pub const #start_ident : usize = #start;
            pub const #end_ident   : usize = #end;
            pub const #range_ident: std::ops::Range<usize> = #start..#end;
        }
    });

    // Final token stream
    TokenStream::from(quote! {
        pub mod #module {
            #![allow(non_upper_case_globals)]

            #( #per_item_consts )*

            pub const TOTAL : usize = #total;
        }
    })
}

/* ──────── parsing helpers ─────────────────────────────────────────────── */

struct LayoutDef {
    module: Ident,
    items: Vec<Item>,
}

impl Parse for LayoutDef {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // `mod` keyword + module ident
        input.parse::<Mod>()?;
        let module: Ident = input.parse()?;

        // brace-delimited list of `Name : len ,`
        let content;
        braced!(content in input);
        let items = Punctuated::<Item, Token![,]>::parse_terminated(&content)?
            .into_iter()
            .collect();

        Ok(Self { module, items })
    }
}

struct Item {
    name: Ident,
    _colon: Token![:],
    len: LitInt,
}

impl Parse for Item {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Item {
            name: input.parse()?,
            _colon: input.parse()?,
            len: input.parse()?,
        })
    }
}
