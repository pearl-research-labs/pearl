import { ArrowLeft } from "lucide-react";
import type { PropsWithChildren } from "react";

type HeaderProps = PropsWithChildren & {
	onBack: () => void;
	name: string;
};
const Header = (props: HeaderProps) => {
	return (
		<div className="flex flex-shrink-0 justify-center border-b border-gray-200 bg-white/80 shadow-sm backdrop-blur-sm">
			<div className="flex w-full max-w-5xl items-center gap-4 px-5 py-6 sm:px-8">
				<button
					type="button"
					onClick={props.onBack}
					className="rounded-lg p-2 transition-colors hover:bg-gray-100"
				>
					<ArrowLeft className="h-5 w-5 text-gray-700" />
				</button>

				<h1 className="text-xl font-semibold text-gray-900">{props.name}</h1>
			</div>
		</div>
	);
};

export { Header };
